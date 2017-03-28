\version "2.10.33"

\header {
  title = "289 - Nun ruhen alle Wälder"
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
  f4 f4 c4 d8 c8 
  bes2 a4 f4 
  bes4 c4 d8 ees16 f16 ees8 d8   %5
  c2 r4 d4 
  bes4 c4 d8 ees8 f4 
  ees2 d4 d8 ees8 
  f4 f4 c4 d4 
  bes2 a4 f4   %10
  bes4 c4 d4 c8 d16 ees16 
  d4 c4 bes4 
  
}

alto = \relative c'' {
  \partial 4 bes4 
    g4 a4 bes4 bes4 
  bes4 a4 bes4 f4 
  bes4 bes4. a8 a4 
  g2 f4 c4 
  f4 g8 a8 bes4 bes4   %5
  f2 r4 a4 
  a8 g8 g8 f8 f8 g8 aes8 g8 
  g4 fis4 g4 bes4 
  c8 a8 bes4 g8 f8 a4~ 
  a4 g4 f4 c4   %10
  f4 ees4 f4 g4 
  f4 f4 f4 
  
}

tenor = \relative c' {
  \partial 4 f4 
    ees4 ees4 f4 f4 
  g4 f4 f4 bes,8 c8 
  d8 ees8 f4 f4 f8 ees8 
  d4 c4 c4 a4 
  bes8 d8 ees4 f8 d8 bes4   %5
  a2 r4 d4 
  d8 ees8 ees8 f8 d2 
  g,4 a4 b4 g'4 
  f8 ees8 d4 e8 f8 f4 
  f4 e4 c4 a4   %10
  bes4. a8 bes4 bes4 
  bes4 a8 ees'8 d4 
  
}

baixo = \relative c {
  \partial 4 bes4 
    ees4 d8 c8 bes4 d4 
  c4 f4 bes,4 bes'4 
  bes,8 c8 d8 ees8 f4 d4 
  g8 f8 e8 c8 f4 f8 ees8 
  d4 c4 bes4 ees4   %5
  f2 r4 fis4 
  g4 a4 bes4 b4 
  c4 c,4 g'4 g4 
  a8 f8 bes4. a8 g8 f8 
  g4 c,4 f4 f8 ees8   %10
  d4 c4 bes4 ees4 
  f4 f,4 bes4 
  
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