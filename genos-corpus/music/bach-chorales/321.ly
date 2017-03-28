\version "2.10.33"

\header {
  title = "321 - Wir Christenleut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \partial 4 g8  a8 
    bes4 a4 g4 g4 
  bes4 a4 g4 d'4 
  c4 bes4 a4 a4 
  bes4 bes4 c4 c4 
  d4 d4 c4 bes4   %5
  a2 g4 d'4 
  c4 bes4 a4 d4 
  c4 bes4 a4 a4 
  bes4 bes4 c4 c4 
  d4 d4 c4 bes4   %10
  a2 g4 
  
}

alto = \relative c' {
  \partial 4 d8  fis8 
    g4 fis4 g4 ees4 
  e4 fis4 d4 d8 e8 
  f8 ees8 d8 e8 fis4 fis4 
  g4 g4 g4 a4 
  bes4 bes4 a4 g4   %5
  g4 fis4 g4 f4 
  f4 d4 d4 f4 
  g4 f8 e8 f4 e4 
  fis4 g4 g4 a4 
  a8 c8 bes8 a8 g8 fis8 g4   %10
  g4 fis4 d4 
  
}

tenor = \relative c' {
  \partial 4 bes8  c8 
    d4 d8 c8 bes4 c4 
  g4 d'4 bes4 f8 g8 
  a4 bes8 c8 d4 d4 
  d4 ees4 ees8 d8 c4 
  f4 g4 ees4 d4   %5
  ees4 d8 c8 bes4 bes4 
  a4 g4 fis4 bes4 
  bes8 g8 c4 c4 cis4 
  d8 c8 bes8 a8 g8 g'8 f8 ees8 
  d4 d4 g,8 a8 bes8 g8   %10
  e'4 d8 c8 b4 
  
}

baixo = \relative c' {
  \partial 4 bes8  a8 
    g4 d4 ees4 c4 
  cis4 d4 g,4 bes4 
  f'4 g4 d4 d4 
  g8 f8 ees8 d8 c4 f8 ees8 
  d8 c8 bes8 a8 g8 fis8 g4   %5
  c4 d4 g,4 bes4 
  f'4 g4 d4 bes'4 
  e,4 c4 f4 a4 
  d,4 ees4 e4 f4 
  fis4 g4 ees4 d4   %10
  cis4 d4 g,4 
  
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