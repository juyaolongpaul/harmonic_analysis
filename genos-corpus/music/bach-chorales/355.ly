\version "2.10.33"

\header {
  title = "355 - Nun ruhen alle Wälder"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 d4 
    bes4 c4 d8 ees8 f4 
  ees2 d4 d8 ees8 
  f4 f4 c4 d4 
  bes2 a4 f4 
  bes4 c4 d4 d8 c8   %5
  c2. d4 
  bes4 c4 d8 ees8 f4 
  ees2 d4 d4 
  f4 f4 c4 d4 
  bes2 a4 f4   %10
  bes4 c4 d4 ees4 
  d8 c8 c4 bes4 
  
}

alto = \relative c' {
  \partial 4 f4 
    ees4 ees4 d8 c8 bes4 
  bes'4 f4 f4 g4 
  f8 ees8 f8 g8 a4 a4 
  g2 c,4 c4 
  f4 g8 a8 bes4 f4   %5
  f2. a4 
  g4 f4 f4 g4 
  g4 fis4 g4 g4 
  f4 f4 f4 f4 
  f4 e4 f4 a4   %10
  g4 f4 f4 bes8 a8 
  bes4 a4 f4 
  
}

tenor = \relative c' {
  \partial 4 bes4 
    g4 a4 bes8 c8 d4 
  d4 c4 bes4 bes4 
  bes4 bes4 f'4 f4 
  f4 e4 f4 a,4 
  bes4 ees4 f8 a,8 bes4   %5
  a2. d4 
  d4 c4 bes8 c8 d4 
  c8 b8 a4 b4 bes4 
  a8 g8 a8 bes8 c4 bes8 c8 
  d4 c4 c4 d4   %10
  d4 c4 bes4 bes8 c8 
  d8 bes8 f'8 ees8 d4 
  
}

baixo = \relative c {
  \partial 4 bes4 
    ees8 d8 c4 bes4 bes'8 a8 
  g4 a4 bes4 g4 
  d8 c8 d8 ees8 f4 d4 
  g2 f4 ees4 
  d4 c4 bes8 c8 d8 ees8   %5
  f2. fis4 
  g4 a4 bes4 b4 
  c4 c,4 g'4 g4 
  d'4 c8 bes8 a4 bes8 a8 
  g4 c4 f,4 d4   %10
  g4 a4 bes8 a8 g4 
  f8 ees8 f4 bes,4 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}