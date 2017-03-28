\version "2.10.33"

\header {
  title = "94 - Warum betrübst du dich, mein Herz"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    g8 a8 bes4 a4 d8 d8 
  c4 bes4 a4 bes8 a8 
  g4 d'4 c4 d4 
  g,8 a8 b4 c4 bes4 
  a4 d4 c4 bes4   %5
  a2 r4 g8 a8 
  bes4 bes4 c4 c4 
  d4 d4 bes4 d4 
  c4 bes4 a4 g8 a8 
  bes4 a g   %10
  
}

alto = \relative c' {
  \partial 4 d4 
    ees4 d8 e8 fis4 g8 g8 
  a4 g4 fis4 g4 
  g4 f8 g8 a4 b4 
  c4 g8 g8 g4 g4 
  fis4 g4 a4 d,4   %5
  d2 r4 d4 
  g4 g8 f8 ees4 f8 g8 
  aes4 g8 f8 g4 g4 
  g8 fis8 g4 fis4 g4 
  g4 fis d   %10
  
}

tenor = \relative c' {
  \partial 4 g4 
    c4 bes8 c8 d4 d8 d8 
  d4 d4 d4 d8 c8 
  bes8 c8 d8 ees8 f4 f4 
  g4 d8 d8 ees4 d4 
  d4 d4 fis,4 g4   %5
  fis2 r4 bes8 c8 
  d4 des4 c8 bes8 aes8 g8 
  f4 f'4 ees4 f4 
  ees4 e4 a,8 bes8 c4 
  d8 e8 a,8 c8 b4   %10
  
}

baixo = \relative c {
  \partial 4 b4 
    c8 c8 g4 d'4 bes'8 bes8 
  fis4 g4 d4 g,4 
  g'8 a8 bes4 f4 d4 
  ees8 f8 g4 c,4 g'4 
  d8 c8 bes4 a4 g4   %5
  d'2 r4 g,4 
  g'8 f8 ees4 aes8 g8 f4 
  bes,4 bes4 ees4 b4 
  c4 cis4 d4 ees4 
  d8 cis8 d4 g,  %10
  
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