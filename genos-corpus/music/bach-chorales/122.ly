\version "2.10.33"

\header {
  title = "122 - Ist Gott mein Schild und Helfersmann"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    c4 b4 c4 d8 ees8 
  bes4. aes8 g4 g4 
  f4 bes4 g4 c4 
  b4. c8 d4 ees4 
  d8 c8 bes8 aes8 g4 aes8 bes8   %5
  f2 ees4 ees'4 
  d4 c8 bes8 bes4 c4 
  bes4 aes8 g8 g4 f4 
  bes4. aes8 g4 ees'8 d8 
  b4. c8 d4 ees4   %10
  f8 ees8 d8 c8 b4 c8 d8 
  d4. c8 c4 ees4 
  ees8 d8 c8 b8 g'4 c,8 a8 
  b4. c8 c2 
  
}

alto = \relative c' {
  \partial 4 ees8  f8 
    g4 g4 g4 f8 ees8 
  f4 d4 ees4 ees4 
  c4 bes8 f'8 ees8 d8 c8 g'8 
  f8 ees8 ees8 g16 fis16 g4 g4 
  f4. d8 ees4 ees4   %5
  ees4 d4 bes4 g'4 
  f4 ees8 f8 g4 aes8 g8 
  f8 ees8 f8 d8 ees4 d4 
  e4 f4 d4 g4 
  f8 d8 ees8 g16 fis16 g4 g4   %10
  f8 g8 aes4 aes4 g8 aes8 
  g4 f8 ees8 ees4 aes4 
  aes4 aes8 g8 g4 g8 f8 
  d8 ees16 f16 g4 g2 
  
}

tenor = \relative c' {
  \partial 4 c4 
    c4 d4 ees4 g,4 
  f4 bes4 bes4 bes4 
  aes8 g8 f8 bes8 bes8 g'8 f8 ees8 
  d4 g,8 a8 b4 c8 bes8 
  aes8 f8 bes4 bes4 aes8 g8   %5
  c4 bes8 aes8 g4 bes4 
  aes8 bes8 c8 d8 ees4 aes,4 
  bes8 c8 d4 bes4 bes4 
  bes4 c4 b4 c4 
  d4 g,8 a8 b4 c4   %10
  c4 f4 f4 ees8 c8 
  c8 a8 b4 c4 c4 
  c8 f8 ees8 d8 c8 b8 c8 d8 
  g,4 d'4 e2 
  
}

baixo = \relative c {
  \partial 4 c4 
    ees4 g4 c4 c,4 
  d4 bes4 ees4 ees4 
  ees4 d4 ees4 aes4 
  d,4 c4 g'4 c4 
  f,8 ees8 d8 bes8 ees8 d8 c8 bes8   %5
  aes4 bes4 ees4 ees4 
  f8 g8 aes4 g4 f8 ees8 
  d8 c8 bes4 ees4 bes'8 aes8 
  g4 f4 f4 ees4 
  d4 c4 g'4 c8 bes8   %10
  aes8 g8 f8 ees8 d4 ees8 f8 
  g4 g,4 aes4 aes'8 g8 
  f4 f4 ees8 d8 ees8 f8 
  g4 g,4 c2 
  
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